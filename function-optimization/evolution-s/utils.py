import os
import sys

def outputPrint(i):
    sys.stdout = open('output_' + i + ".txt", 'w')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
