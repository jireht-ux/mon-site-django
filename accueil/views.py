
from django.shortcuts import render
from datetime import date
from .forms import NomForm

def index(request):
    message = "Bienvenue sur mon site Django !"
    nom = None

    if request.method == "POST":
        form = NomForm(request.POST)
        if form.is_valid():
            nom = form.cleaned_data['nom']
            message = f"Bonjour, {nom} !"
    else:
        form = NomForm()

    contexte = {
        'message': message,
        'date_du_jour': date.today(),
        'form': form
    }
    return render(request, 'accueil/index.html', contexte)