import json
import nltk
from nltk.tokenize import sent_tokenize
import re

def preprocess_text(text:str):

    # can be used PR desc, commit message, commentsm - general preprocessing rules

    text = re.sub(r'\\r', '', text)
    text = re.sub(r'\\n', r'\n', text)

    # Replace two \n with full stop
    text = re.sub(r'\n{2,}', r'. ', text)

    # Replace one \n with space
    text = re.sub(r'\n', r' ', text)

    # Replace more than on full stop with one full stop
    text = re.sub(r'\.{2,}', r'.', text)

    # Replace more than one space with one space
    text = re.sub(r'\s{2,}', r' ', text)

    # Remove the leading and trailing spaces
    text = text.strip()
    
    # Convert text to lowercase
    # text = text.lower()
    
    sent_list = sent_tokenize(text)
    proc_list = []

    # eliminate the sentences with the given patterns
    url_pattern = r'https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))'
    structure_pattern = r'^#+'
    signature_pattern = r'^(signed-off-by|co-authored-by|also-by):'
    email_pattern = r'(^|\s)<[\w.-]+@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]>'
    at_pattern = r'@\S+'
    reference_pattern = r'#[\d]+'

    pattern_list = [email_pattern, url_pattern, reference_pattern, signature_pattern, structure_pattern, at_pattern]

    for sent in sent_list:

        # If we find a pattern, we don't add the sentence to the list
        for pattern in pattern_list:
            if re.search(pattern, sent):
                break
        else:
            proc_list.append(sent)

    # concatenate all the sentences
    proc_text = ' '.join(proc_list)
    
    # Word tokenize
    proc_text = nltk.word_tokenize(proc_text)
    proc_list = []

    for word in proc_text:
        version_pattern = r'(^|\s|-)[\d]+(\.[\d]+){1,}' # Replace with 'version'
        sha_pattern = r'(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))' # Replace with 'sha'
        digit_pattern = r'(^|\s|-)[\d]+(?=(\s|$))' # Replace with 0

        pattern_list = [version_pattern, sha_pattern, digit_pattern]

        for pattern in pattern_list:
            if re.search(pattern, word):
                if pattern == version_pattern:
                    word = re.sub(pattern, ' version ', word)
                elif pattern == sha_pattern:
                    word = re.sub(pattern, ' sha ', word)
                elif pattern == digit_pattern:
                    word = re.sub(pattern, ' 0 ', word)
        
        proc_list.append(word)
    
    return ' '.join(proc_list).strip().lower()

def get_comments(comments: list):
    ''' 
        comments: List of comments
    '''
    # Removes the initial '+' and space
    comments = [comment[1:] for comment in comments]
    
    # Remove spaces
    comments = [comment.strip() for comment in comments]

    proc_comments = []
    
    # Extract comments from /**/ and //
    flag = 0
    temp = ""
    for comm in comments:
        if comm.startswith('//'):
            proc_comments.append(comm[2:].strip())
        
        if comm.startswith('/*'):
            flag = 1
            temp = ""
        
        elif comm.startswith('*/'):
            flag = 0
            proc_comments.append(temp.strip())
            temp = ""

        elif comm.startswith('*'):
            if flag == 1:
                temp = temp + comm[1:] + " "
            else:
                proc_comments.append(comm[1:].strip())
        

    return proc_comments

def process_commits(commits):

    '''    
    copy-right comments, license comments, function signatures in Javadocs 
    (e.g., “@param: param1”) and the comments with only
    punctuation marks were filtered.

    concatentate the remaining comments as a comment paragraph

    apply general text preprocessing
    '''

    for commit in commits.values():
        commit['cm'] = preprocess_text(commit['cm'][1:-1])
        comments = get_comments(commit['comments'])

        proc_comments = []
        for comment in comments:
            if re.search(r'copyright', comment):
                continue
            if re.search(r'Copyright', comment):
                continue
            if re.search(r'license', comment):
                continue
            if re.search(r'License', comment):
                continue
            if re.search(r'^[^\w\s]+$', comment):
                continue

            proc_comments.append(comment)

        # commit['comments'] = proc_comments
        comments_para = ' '.join(proc_comments)

        commit['comments'] = preprocess_text(comments_para)
        
    return commits

    

    
if __name__ == '__main__':

    dataset_raw = json.load(open('dataset.json', 'r'))
    dataset_new = dict()

    i = 0
    for d_key in dataset_raw:

        print(f'---------- datapoint {i} --------------')
        i += 1

        # preprocess the PR desc [1:-1] to remove the quotes
        dataset_raw[d_key]['body'] = preprocess_text(dataset_raw[d_key]['body'][1:-1])
        
        dataset_raw[d_key]['commits'] = process_commits(dataset_raw[d_key]['commits'])
 
        # Empty Description PRs
        if len(dataset_raw[d_key]['body']) == 0:
            continue

        # empty or trivial or long descriptions
        # if len(dataset_raw[d_key]['body']) < 5 or len(dataset_raw[d_key]['body']) > 100:
        #     continue

        words = nltk.word_tokenize(dataset_raw[d_key]['body'])
        if len(words) < 5 or len(words) > 100:
            continue

        # Continue if body has only punctuation characters in it
        if re.search(r'^[^\w\s]+$', dataset_raw[d_key]['body']):
            continue

        # empty or trivial or long commits
        if len(dataset_raw[d_key]['commits']) < 2 or len(dataset_raw[d_key]['commits']) > 20:
            continue

        # save the processed data point
        dataset_new[d_key] = dataset_raw[d_key]

    # Save the filtered dataset
    json.dump(dataset_new, open('dataset_filtered.json', 'w+'))