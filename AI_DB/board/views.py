from django.shortcuts import render
from .models import *

# Create your views here.

def index(request):

    pcb1 = PcbImtImg.objects.all()
    pcb2 = PcbImtLabel.objects.all()

    #print(pcb2.label_parts)
    return render(request, 'index.html', {'board_list':pcb1, 'label_list':pcb2})