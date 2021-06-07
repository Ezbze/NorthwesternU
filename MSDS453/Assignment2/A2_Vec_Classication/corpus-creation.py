"""
Created on Fri Apr 24 21:40:09 2020

@author: Ezana.Beyenne
This is the main file
"""
import os  # operating system commands
import json  # JSON utilities from the Python Standard Library
from threaded_web_crawler import ThreadedWebCrawler


# function for walking and printing directory structure
def list_all(current_directory):
    for root, dirs, files in os.walk(current_directory):
        level = root.replace(current_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def writeToDirectory(page_dirname, text, filename, fileType = '.html'):
    # prevent .html.html or .html.txt file extensions
    if '.html' in filename:
        filename = filename.replace('.html','')
        
    if '.htm' in filename:
        filename = filename.replace('.htm','')
    if len(text) > 0:
        with open(os.path.join(page_dirname, filename + fileType), 'w',encoding='utf8') as f:
                f.write(text)
        f.close()
    
def write_to_json_file(corpusdtos, jl_filename):
     data = []
     for obj in corpusdtos:
        index = corpusdtos.index(obj)
        data.append({'ID': index,
                    'URL': obj.url,
                    'TITLE': obj.title,
                    'FILENAME': obj.filename,
                    'BODY': obj.text})
     
     with open(jl_filename, 'w') as f:
        f.write(json.dumps(data))
        f.write('\n')
     f.close()
     
def crawl_search_urls(url, rest_url , pageNumber, run_additionals = False):
    corpus_objects = []
    
    for i in range(pageNumber):
        search_url = url + str(i + 1) + rest_url
        print(search_url)
        
        # Run additionals on the 1st iteration
        if run_additionals == True and (i + 1) == 1:
            run_additionals = True
            
        s = ThreadedWebCrawler(search_url, run_additionals)
        s.run_corpus_scraper()
        
        for corpus in s.corpus_objects:
            corpus_objects.append(corpus)
        
    return corpus_objects

 
if __name__ == '__main__':
    
    page_dirname_raw = 'raw_data'
    if not os.path.exists(page_dirname_raw):
	         os.makedirs(page_dirname_raw)  
             
    page_dirname_corpus = 'corpus'
    if not os.path.exists(page_dirname_corpus):
	         os.makedirs(page_dirname_corpus)
    
    default_page_num = 32
    corpus_dto_objects = []
    
    print('\nStarting Web Crawling...\n')
    wired_search_url ='https://www.wired.com/search/?q=self%20driving%20cars%20safety&page='
    wired_search_args = '&sort=score'
    wired_corpusList = crawl_search_urls(wired_search_url, wired_search_args, default_page_num, True)
    
    for corpus in wired_corpusList:
        corpus_dto_objects.append(corpus)
        
        
        
    verge_search_url ='https://www.theverge.com/search?page='
    verge_search_args = '&q=autonomous+vehicles&type=Article'
    verge_corpusList = crawl_search_urls(verge_search_url,verge_search_args, default_page_num)
    
    for corpus in verge_corpusList:
        corpus_dto_objects.append(corpus)
        
    curbed_search_url ='https://www.curbed.com/search?page='
    curbed_search_args = '&q=self-driving+car&type=Article'
    curbed_corpusList = crawl_search_urls(curbed_search_url,curbed_search_args, default_page_num)
    
    for corpus in curbed_corpusList:
        corpus_dto_objects.append(corpus)
    
   
    
    #print('\nWriting files to raw and corpus folders')
    for obj in corpus_dto_objects:
          writeToDirectory(page_dirname_raw, obj.html, obj.filename)
          writeToDirectory(page_dirname_corpus, obj.text, obj.filename, '.txt')
    
    jil_file_name = 'autonomous_vehicles_safety_corpus.jl'
    print(jil_file_name + ' is being created\n')
    write_to_json_file( corpus_dto_objects, jil_file_name)


    # Print directory structure
    #current_directory = os.getcwd()
    #list_all(current_directory)
     
    print('\nEnd of Web Crawling....')