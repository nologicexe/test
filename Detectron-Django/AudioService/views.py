from django.http import request
from django.http.response import HttpResponseRedirect
from web_project.settings import BASE_DIR, STATIC_ROOT
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from .prediction_module import Process_File

from datetime import datetime
import os

# Create your views here.
from django.http import HttpResponse

@login_required
def home(request):
    return HttpResponse("You can access AudioService results for certain video at /AudioService/<filename>")

@login_required
def results(request, filename):
    return render(
        request,
        'AudioService/results.html',
        {
            'filename': filename,
            'date': datetime.now(),
            'static': STATIC_ROOT
        }
    )

@login_required
def results_redirected(request, uploaded_filename):
    result_plot_name = uploaded_filename.replace(uploaded_filename[-4:], ".png")
    return render(
        request,
        'AudioService/results_redirected.html',
        {
            'uploaded_filename': uploaded_filename,
            'date': datetime.now(),
            'result_plot_name': result_plot_name,
            'static': STATIC_ROOT
        }
    )

@login_required
def upload_file(request):
    if request.method == 'POST' and 'myfile' in request.FILES:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(os.path.join(BASE_DIR, 'AudioService', 'static', 'AudioService'))
        uploaded_filename = fs.save(myfile.name, myfile)
        Process_File(uploaded_filename)
        return HttpResponseRedirect(uploaded_filename)
    return render(request, 'AudioService/upload.html')

