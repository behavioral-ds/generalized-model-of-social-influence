import asyncio
import random
import time
import pickle as pk
import boto3
import xmltodict
import logging
import json
import uuid
import time
import sys

sys.path.insert(1, './code/mturk-emoji/')

from encode_emoji import replace_emoji_characters
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.DEBUG, filename='./data/00_logs/main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

qsort_logger = setup_logger('qsort', './data/00_logs/qsort.log')
comparison_logger = setup_logger('comparison', './data/00_logs/comparison.log')
mturk_logger = setup_logger('mturk', './data/00_logs/mturk.log')
batcher_logger = setup_logger('batcher', './data/00_logs/batcher.log')
banner_logger = setup_logger("banner", "./data/00_logs/banner.log")


create_hits_in_production = True
pct_threshold = 0.2
accuracy_threshold = 0.5
volume_threshold = 100 

environments = {
  "production": {
    "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
    "preview": "https://www.mturk.com/mturk/preview"
  },
  "sandbox": {
    "endpoint": 
          "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
    "preview": "https://workersandbox.mturk.com/mturk/preview"
  },
}
mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]
session = boto3.Session(profile_name='mturk')
client = session.client(
    service_name='mturk',
    region_name='us-east-1',
    endpoint_url=mturk_environment['endpoint'],
)

#with open('./data/03_processed/09092020_mini.pk', 'rb') as rf:
with open('./data/01_raw/arson_sample_w_tweets.pk', 'rb') as rf:
    targets, proxies, sample = pk.load(rf)

question_pool = ['Which user is the proxy user most likely to retweet?',
                 'Who will the proxy user be more socially influenced by?',
                 'Which user would sway the proxy user\'s opinion more?']

questions_per_HIT = 10 if create_hits_in_production else 10
batcher_retry_count = 20 if create_hits_in_production else 3

qr = [{'QualificationTypeId':'000000000000000000L0',
                        'Comparator': 'GreaterThan',
                        'IntegerValues':[90]},
    {'QualificationTypeId':'00000000000000000040',
                        'Comparator': 'GreaterThan',
                        'IntegerValues':[200]},
    {'QualificationTypeId':'3XDD2ODWMUQYXMV9MO69YQGBQXBGXK',
                        'Comparator': 'DoesNotExist'}
    ]

TaskAttributes = {
    # How many times can a worker, see a question
    'MaxAssignments': 1,
    # How long the task will be available on MTurk (24 hour)
    'LifetimeInSeconds': 24*60*60,
    # How long Workers have to complete each item (15 minutes)
    'AssignmentDurationInSeconds': 60*15,
    # The reward you will offer Workers for each response
    'Reward': '0.04',
    'Title': 'Compare the influence of Twitter Users',
    'Keywords': 'twitter, influence, comparison, pairwise',
    'Description': 'Pairwise compare the influence of 10 Twitter User Pairs.',
    'QualificationRequirements':qr
}

batches_sent = []

def getAnswers(batches_sent):
    res_dict = {}
    all_answers = []
    for item in batches_sent:
        hit = client.get_hit(HITId=item['hit_id'])
        item['status'] = hit['HIT']['HITStatus']
        # Get a list of the Assignments that have been submitted
        assignmentsList = client.list_assignments_for_hit(
            HITId=item['hit_id'],
            AssignmentStatuses=['Submitted', 'Approved'],
            MaxResults=10
        )
        assignments = assignmentsList['Assignments']
        item['assignments_submitted_count'] = len(assignments)
        answers = []
        for assignment in assignments:

            # Retreive the attributes for each Assignment
            worker_id = assignment['WorkerId']
            assignment_id = assignment['AssignmentId']

            # Retrieve the value submitted by the Worker from the XML
            answer_dict = xmltodict.parse(assignment['Answer'])
            answer = answer_dict['QuestionFormAnswers']['Answer']['FreeText']
            answers.append({'worker_id':worker_id, 'comparisons': json.loads(answer)})

            # Approve the Assignment (if it hasn't been already)
            if assignment['AssignmentStatus'] == 'Submitted':
                client.approve_assignment(
                    AssignmentId=assignment_id,
                    OverrideRejection=False
                )

        # Add the answers that have been retrieved for this item
        item['answers'] = answers
        all_answers += answers
    
    flattened = [{'worker_id': obj['worker_id'], 'comparisons': x} for obj in all_answers for x in obj['comparisons']]
    worker_accuracies = defaultdict(lambda : defaultdict(int))
    banner_logger.info(str(flattened))
    for bat in flattened:
        for k,v in bat['comparisons'].items():
            section_id, proxy, left, right = k.split('_*_')
            labels = (left, right)
            loc_res = list(v.values())[0]
            res_dict[labels] = loc_res
            left_user, right_user = sample.loc[left], sample.loc[right]
            banner_logger.info(str(left_user['pct']))
            follower_difference = left_user['pct'] - right_user['pct']
            banner_logger.info("Follower difference was: "+str(follower_difference)+" with threshold: "+str(pct_threshold))
            if np.abs(follower_difference) >= pct_threshold: 
                if follower_difference > 0 == loc_res:
                    # Is correct
                    worker_accuracies[bat['worker_id']]['correct'] += 1
                else:
                    worker_accuracies[bat['worker_id']]['incorrect'] += 1
    banner_logger.info("Worker acuracy dict is: "+str(worker_accuracies))
    for worker_id, obj in worker_accuracies.items():
        worker_accuracies[worker_id]['volume'] = worker_accuracies[worker_id]['correct'] + worker_accuracies[worker_id]['incorrect']
        worker_accuracies[worker_id]['accuracy'] = worker_accuracies[worker_id]['correct']/worker_accuracies[worker_id]['volume'] 
        banner_logger.info("Computed accuracy of: " +str(worker_id)+" with accuracy, "+str(worker_accuracies[worker_id]['accuracy'])+" with volume, "+str(worker_accuracies[worker_id]['volume']))
    return res_dict, worker_accuracies

def mix_batch(trip):
    proxy_id, user_left_id, user_right_id = trip
    return (proxy_id, user_left_id, user_right_id) if random.choice([True, False]) else (proxy_id, user_right_id, user_left_id)


def makeHIT(batch):
    batch = [mix_batch(trip) for trip in batch]
    failure_flag = True
    number_of_recent_tweets = 5
    while failure_flag:
        try:
            question_xml = makeHIT_XML(batch, number_of_recent_tweets)
            mturk_logger.info('Size of xml is '+str(sys.getsizeof(question_xml)))
            response = client.create_hit(
                **TaskAttributes,
                Question=question_xml
            )
            failure_flag = False
        except Exception as e:
            mturk_logger.info(str(e))
            mturk_logger.info('Reducing number of recent tweets to '+str(number_of_recent_tweets-1))
            if number_of_recent_tweets > 0:
                number_of_recent_tweets -= 1
            else:
                raise Exception(e)
    new_batch = {
        'batch': batch,
        'hit_id': response['HIT']['HITId'],
        'hit_type_id' :response['HIT']['HITTypeId']
    }
    batches_sent.append(new_batch)
    mturk_logger.info('Sent the Following:')
    mturk_logger.info(str(new_batch))

def makeHIT_XML(batch, number_of_recent_tweets):
    QUESTION_XML = """<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
        <HTMLContent><![CDATA[{}]]></HTMLContent>
        <FrameHeight>650</FrameHeight>
        </HTMLQuestion>"""
    form_layout = open('./code/html/form.html', 'r').read()
    sections = ''
    for i in range(len(batch)):
        proxy_id, user_left_id, user_right_id = batch[i]
        sections += makeSectionHTML(str(i), str(proxy_id), str(user_left_id), str(user_right_id), number_of_recent_tweets=number_of_recent_tweets)
        sections += '\n'
    form_layout = form_layout.replace('{{content}}', sections)
    question_xml = QUESTION_XML.format(form_layout)
    return question_xml


def makeSectionHTML(section_id, proxy_id, user_left_id, user_right_id, number_of_recent_tweets=5):
    section_layout = open('./code/html/section.html', 'r').read()
    question = question_pool[random.randint(0,len(question_pool)-1)]
    
    user_proxy = makeUserHTML(proxy_id, number_of_recent_tweets=number_of_recent_tweets)
    user_left = makeUserHTML(user_left_id, number_of_recent_tweets=number_of_recent_tweets)
    user_right = makeUserHTML(user_right_id, number_of_recent_tweets=number_of_recent_tweets)
    
    user_pname = proxy_id
    user_lname = user_left_id
    user_rname = user_right_id
    
    section_layout = section_layout.replace('{{question}}', question)
    section_layout = section_layout.replace('{{user_proxy}}', user_proxy)
    section_layout = section_layout.replace('{{user_left}}', user_left)
    section_layout = section_layout.replace('{{user_right}}', user_right)
    section_layout = section_layout.replace('{{section_id}}', section_id)
    section_layout = section_layout.replace('{{user_pname}}', user_pname)
    section_layout = section_layout.replace('{{user_lname}}', user_lname)
    section_layout = section_layout.replace('{{user_rname}}', user_rname)
    
    return (section_layout)

def makeUserHTML(user_id, number_of_recent_tweets=5):
    user_layout = open('./code/html/user.html', 'r').read()
    user = sample.loc[user_id]
    screen_name = user_id
    name = user['name']
    description = user['description'] if user['description'] != '' else '...'
    url = "https://www.twitter.com/"+screen_name
    image = user['url']
    follower_count = str(user['followers_count'])
    friend_count = str(user['friends_count'])
    statuses_count = str(user['statuses_count'])
    user_layout = user_layout.replace('{{user_url}}', url)
    user_layout = user_layout.replace('{{screen_name}}', screen_name)
    user_layout = user_layout.replace('{{name}}', replace_emoji_characters(name))
    user_layout = user_layout.replace('{{description}}', replace_emoji_characters(description))
    user_layout = user_layout.replace('{{user_image}}', image)
    user_layout = user_layout.replace('{{follower_count}}', follower_count)
    user_layout = user_layout.replace('{{friend_count}}', friend_count)
    user_layout = user_layout.replace('{{statuses_count}}', statuses_count)
    user_layout = user_layout.replace('{{user_tweets}}', makeTweetsHTML(user_id, number_of_recent_tweets))
    return user_layout

def makeTweetsHTML(user_id, number_of_recent_tweets=5):
    user = sample.loc[user_id]
    tweets = ''
    for tweet, t in user['recent_tweets'][:number_of_recent_tweets]:
        tweets += makeTweetHTML(tweet[:100], t)
        tweets += '\n'
    return tweets

def makeTweetHTML(tweet, t):
    tweet_layout = open('./code/html/tweet.html', 'r').read()
    tweet_layout = tweet_layout.replace('{{tweet_content}}', replace_emoji_characters(tweet))
    tweet_layout = tweet_layout.replace('{{tweet_time}}', str(t))
    return tweet_layout

def banWorkers(worker_accuracies):
    global worker_banned_dict
    for worker_id, v in worker_accuracies.items():
        volume = v['volume']
        accuracy = v['accuracy']
        if accuracy <= accuracy_threshold and volume >= volume_threshold and worker_id not in worker_banned_dict:
                banner_logger.info("Banned worker: " +str(worker_id)+" with accuracy, "+str(accuracy)+" with volume, "+str(volume))
                client.associate_qualification_with_worker(QualificationTypeId='3XDD2ODWMUQYXMV9MO69YQGBQXBGXK', WorkerId=worker_id, IntegerValue=volume)
                worker_banned_dict[worker_id] = accuracy
        elif volume >= volume_threshold and worker_id not in worker_banned_dict:
            worker_banned[worker_id] = accuracy
            banner_logger.info("Did not ban worker: " +str(worker_id)+" with accuracy, "+str(accuracy)+" with volume, "+str(volume))

async def result_writer():
    #Check's if there is a new result
    global is_done, results_dict, worker_accuracy_dict
    while not is_done:
        # print('In Result Writer...')
        await asyncio.sleep(10)
        res_dict, worker_accuracy_dict = getAnswers(batches_sent)
        banWorkers(worker_accuracy_dict)
        results_dict = res_dict
        if len(results_dict.keys() - results_check.keys()) > 0:
            # print(results_dict.keys() - results_check.keys())
            results_event.set()
        else:
            results_event.clear()


async def batcher():
    # We need to put some kind of retry count in here, so we don't stall at the end
    # We could also make sure in some way that questions aren't the same for any batch (we might use one set of queues per qsort)
    retry_count = 0
    global is_done
    while not is_done:
        if compQ.qsize() >= questions_per_HIT or retry_count >= batcher_retry_count:
            if compQ.qsize() > 0:
                if retry_count >= batcher_retry_count:
                    batcher_logger.info('Failure with size: '+str(compQ.qsize()))
                retry_count = 0
                batch = [(list(proxies.index)[random.randint(0,len(proxies.index)-1)],*(await compQ.get())) for _ in range(min(questions_per_HIT,compQ.qsize()))]
                makeHIT(batch)
            else:
                retry_count = 0
        else:
            retry_count += 1
            batcher_logger.info(str(compQ.qsize()))
            await asyncio.sleep(5)

async def compare(i,j):
    comparison_logger.info(f'Comparing {i} and {j}')
    await compQ.put((i,j))
    global results_dict, results_check
    while (i,j) not in results_dict:
        comparison_logger.info(f'Checking if {i} and {j} are in the results dict')
        await results_event.wait()
        await asyncio.sleep(1)
    local_result = results_dict[(i,j)]
    comparison_logger.info(f'Acquiring lock for {i} and {j}')
    await results_lock.acquire()
    comparison_logger.info(f'Acquired lock for {i} and {j}')
    results_check[(i,j)] = True
    results_check[tuple(reversed((i,j)))] = True
    results_lock.release()
    comparison_logger.info(f'Released lock for {i} and {j}, with local results {local_result}')
    return (i,j , local_result)

async def qsort(arr):
    qsort_logger.info('In qsort with arr ' +str(arr))
    if len(arr) <= 1:
        return arr
    p = arr[0]
    l = list()
    u = list()
    futures = [asyncio.ensure_future(compare(e,p)) for e in arr[1:]]
    for e, _, res in await asyncio.gather(*futures):
        if res:
            l += [e]
        else:
            u += [e]
    qsort_logger.info('Finished partitioning array' +str(arr)+' with left: ' + str(l) + f' and pivot [{p}] and right: ' +str(u) )
    l = asyncio.ensure_future(qsort(l))
    u = asyncio.ensure_future(qsort(u))
    await asyncio.gather(l,u)
    qsort_logger.info('Sorted '+str(l.result() + [p] + u.result()))
    return l.result() + [p] + u.result()


async def qsort_wrapper(arr):
    qsort_logger.info('In qsort_wrapper with arr '+str(arr))
    res = await qsort(arr)
    global is_done
    is_done = True
    return res


targets_to_sort = list(targets.index)
random.shuffle(targets_to_sort)

loop = asyncio.get_event_loop()
compQ = asyncio.Queue(loop=loop, maxsize=500)
HITQ = asyncio.Queue(loop=loop, maxsize=500)
resQ = asyncio.Queue(loop=loop, maxsize=500)
results_event = asyncio.Event(loop=loop)
results_lock = asyncio.Lock(loop=loop)
worker_accuracy_dict = {}
worker_banned_dict = {}
results_dict = {}
results_check = {}
is_done = False
try:
    temp = asyncio.ensure_future(qsort_wrapper(targets_to_sort))
    loop.set_debug(True)
    loop.run_until_complete(asyncio.gather(temp,asyncio.ensure_future(result_writer()),asyncio.ensure_future(batcher())))
    qsort_logger.info(temp.result())
except KeyboardInterrupt:
    pass
finally:
    loop.close()
    filename = time.strftime("%Y%m%d-%H%M%S")
    with open('./data/05_output/results/'+filename+'.pk', 'wb') as wf:
        pk.dump(results_dict, wf)
    with open('./data/05_output/batches_sent/'+filename+'.pk', 'wb') as bsf:
        pk.dump(batches_sent, bsf)
