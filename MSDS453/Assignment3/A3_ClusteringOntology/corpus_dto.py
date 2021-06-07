# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:40:09 2020

@author: Ezana.Beyenne
This is the Data Structure that holds 
the information for information that 
will be sent to the jl file
"""

class Corpus(object):
     def __init__(self, filename, url, text, html, title):
         self.filename = filename
         self.url = url
         self.text = text
         self.html = html
         self.title = title
         