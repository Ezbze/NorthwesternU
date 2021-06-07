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
    
    with open(os.path.join(page_dirname, filename + fileType), 'w',encoding='utf8') as f:
            f.write(text)
    f.close()
    
def write_to_json_file(corpusdtos, jl_filename):
    with open(jl_filename, 'w') as f:
     for obj in corpusdtos:
        index = corpusdtos.index(obj)
        itemdict = {'ID': index,
                    'URL': obj.url,
                    'TITLE': obj.title,
                    'FILENAME': obj.filename,
                    'BODY': obj.text}
        f.write(json.dumps(itemdict))
     f.write('\n')
     f.close()
    
 
if __name__ == '__main__':
    
    page_dirname_raw = 'raw_data'
    if not os.path.exists(page_dirname_raw):
	         os.makedirs(page_dirname_raw)  
             
    page_dirname_corpus = 'corpus'
    if not os.path.exists(page_dirname_corpus):
	         os.makedirs(page_dirname_corpus)
    
    print('\nStarting Web Crawling...\n')
    s = ThreadedWebCrawler("https://www.wired.com/search/?q=self%20driving%20cars%20safety&page=1&sort=score")
    s.run_corpus_scraper()
    
    #print('\nWriting files to raw and corpus folders')
    for obj in s.corpus_objects:
         #print(obj.filename)
         writeToDirectory(page_dirname_raw, obj.html, obj.filename)
         writeToDirectory(page_dirname_corpus, obj.text, obj.filename, '.txt')
    
    jil_file_name = 'autonomous_vehicles_safety_corpus.jl'
    #print(jil_file_name + ' is being created\n')
    write_to_json_file(s.corpus_objects, jil_file_name)


    # Print directory structure
    current_directory = os.getcwd()
    list_all(current_directory)
     
    print('\nEnd of Web Crawling....')