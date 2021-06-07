# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:37:28 2020

@author: Ezana.Beyenne
This handles the scraping, information extraction
"""
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from corpus_dto import Corpus


class ThreadedWebCrawler:
 
    def __init__(self, base_url):
        self.base_url = base_url
        self.corpus_objects = []
        self.root_url = '{}://{}'.format(urlparse(self.base_url).scheme, urlparse(self.base_url).netloc)
        self.pool = ThreadPoolExecutor(max_workers=20)
        self.scraped_pages = set([])
        self.to_crawl = Queue()
        self.to_crawl.put(self.base_url)
        self.additionalUrls = ['https://www.nhtsa.gov/technology-innovation/automated-vehicles-safety',
                               'https://www.ghsa.org/issues/autonomous-vehicles',
                               'https://www.vox.com/recode/2019/5/17/18564501/self-driving-car-morals-safety-tesla-waymo',
                               'https://www.curbed.com/2016/9/21/12991696/driverless-cars-safety-pros-cons',
                               'https://www.curbed.com/transportation/2018/3/20/17142090/uber-fatal-crash-driverless-pedestrian-safety',
                               'https://www.curbed.com/transportation/2018/3/23/17153200/delete-uber-cities',
                               'https://www.rand.org/pubs/research_reports/RR1478.html']
 
    # parse all the urls with the exception of the search
    def parse_links(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.findAll('a', attrs={'class':'archive-item-component__link'}, href=True)
        for link in links:
            url = link['href']
            if url.startswith('/') or url.startswith(self.root_url):
                url = urljoin(self.root_url, url)
                if url not in self.scraped_pages:
                    self.to_crawl.put(url)
                    
        # additional urls 
        for additionalUrl in self.additionalUrls:
            if additionalUrl not in self.scraped_pages:
                 self.to_crawl.put(additionalUrl)
                    
    def text_from_html(self, body, articleBodyCssSelector):
        soup = BeautifulSoup(body, 'html.parser')
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.decompose()    # rip it out
        
        title = soup.find('title')
        items = "".join([item.text for item in soup.select(articleBodyCssSelector)])
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in items.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text, title.text
    
    def parse_url(self, url):
        a = urlparse(url)
        page =Path(a.path).name
    
        filename = '%s' % page 
        if '.html' not in filename:
            filename = filename + '.html'
        
        # www.wired.com -> domain wired
        domainName =a.netloc.split('www.')[-1].split('.')[0]
        #print(filename)
        #print(domainName)
        return filename, domainName
                    
    def determine_css_selector(self, domain):
         # determine what css selector to use to get the article body
         cssSelector = '.article__chunks p' # wired.com css selector
         if domain == 'nhtsa':
              cssSelector = '.article--copy p'
         elif domain == 'ghsa':
              cssSelector = 'article'
         elif domain == 'vox':
              cssSelector = '.c-entry-content p'
         elif domain == 'curbed':
              cssSelector = 'main p'
         elif domain == 'rand':
              cssSelector = '.product-body p'
          
         #print(cssSelector)
         return cssSelector
        
    def scrape_info(self, html, url):
        if '/search/' not in url:
            filename, domain = self.parse_url(url)
            
            # determine what css selector to use to get the article body
            cssSelector = self.determine_css_selector(domain)
              
            text, title = self.text_from_html(html, cssSelector)
            # shorten filename 
            self.corpus_objects.append(Corpus(domain[0:2] + '-' + filename[0:15], url, text, html, title))
        return
 
    def post_scrape_callback(self, res):
        result = res.result()
        if result and result.status_code == 200:
            self.parse_links(result.text)
            self.scrape_info(result.text, result.url)
            #print("...")
 
    def scrape_page(self, url):
        try:
            res = requests.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return
 
    def run_corpus_scraper(self):
        while True:
            try:
                target_url = self.to_crawl.get(timeout=30)
                if target_url not in self.scraped_pages:
                    print("Scraping URL: {}".format(target_url))
                    self.scraped_pages.add(target_url)
                    job = self.pool.submit(self.scrape_page, target_url)
                    job.add_done_callback(self.post_scrape_callback)
            except Empty:
                return
            except Exception as e:
                print(e)
                continue