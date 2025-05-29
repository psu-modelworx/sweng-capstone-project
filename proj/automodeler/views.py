from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect

from .models import *

# Create your views here.

def index(request):
    if request.user.is_authenticated:
        return render(request, "automodeler/index.html", {})
    else:
        url = reverse("login")
        return HttpResponseRedirect(url)