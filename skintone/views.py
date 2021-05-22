from skintone.forms import ImageUploadForm
from django.shortcuts import render
from .forms import ImageUploadForm

import os
import pprint
import sys
import cv2
import imutils
import numpy
from collections import Counter
from matplotlib import pyplot
from sklearn.cluster import KMeans


# Create your views here.
def home(request):
    return render(request, 'skintone/home.html')


def coding(request):
    return render(request, 'skintone/coding.html')


def readmore(request):
    return render(request, 'skintone/readmore.html')


def robust(request):
    return render(request, 'skintone/robust.html')


def thankyou(request):
    return render(request, 'skintone/thankyou.html')


def preview(request):
    return render(request, 'skintone/preview.html')


def capture(request):
    return render(request, 'skintone/capture.html')


def imgprocess(request):
    
    return render(request, 'skintone/thankyou.html')