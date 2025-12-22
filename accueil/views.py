
from django.shortcuts import render
from datetime import date, datetime
from .forms import NomForm
from .models import Tache

def index(request):
    message = "Salut et Bienvenue à tous sur mon super site Django !"
    nom = None

    if request.method == "POST":
        form = NomForm(request.POST)
        if form.is_valid():
            nom = form.cleaned_data['nom']
            message = f"Bonjour, {nom} !"
    else:
        form = NomForm()

    # Détermination du message par défaut selon l'heure (si aucun nom soumis)
    if not nom:
        current_hour = datetime.now().hour
        if current_hour < 12:
            message = "Bon matin !"
        else:
            message = "Bon après-midi !"

    # Récupérer toutes les tâches depuis la base, ordonnées par titre
    # Renvoie un QuerySet (itérable) d'objets Tache
    tasks = Tache.objects.order_by('title')

    contexte = {
        'message': message,
        'date_du_jour': date.today(),
        'form': form,
        'tasks': tasks,
    }
    return render(request, 'accueil/index.html', contexte)