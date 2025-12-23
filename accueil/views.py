
from django.shortcuts import render, redirect
from datetime import date, datetime
from .forms import NomForm, TacheForm
from .models import Tache


def liste_taches(request):
    """Afficher la liste des tâches et les formulaires (GET).

    Cette vue répond aux requêtes GET uniquement. Elle prépare et renvoie :
    - un formulaire vide `TacheForm` pour ajouter une tâche ;
    - un formulaire vide `NomForm` pour la salutation ;
    - la liste des tâches ordonnée par titre ;
    - un message de bienvenue (soit basé sur le nom stocké en session, soit
      une salutation en fonction de l'heure).

    Args:
        request (django.http.HttpRequest): requête HTTP reçue par la vue.

    Returns:
        django.http.HttpResponse: rendu du template `accueil/index.html`.
    """

    # préparer les formulaires non liés
    tache_form = TacheForm()
    nom_form = NomForm()

    # message par défaut selon la session ou l'heure
    nom = request.session.get('nom')
    if nom:
        message = f"Bonjour, {nom} !"
    else:
        current_hour = datetime.now().hour
        if current_hour < 12:
            message = "Bon matin !"
        else:
            message = "Bon après-midi !"

    # récupérer les tâches depuis la base
    try:
        tasks = Tache.objects.order_by('title')
    except Exception:
        tasks = []

    contexte = {
        'message': message,
        'date_du_jour': date.today(),
        'form': nom_form,
        'tache_form': tache_form,
        'tasks': tasks,
    }
    return render(request, 'accueil/index.html', contexte)


def ajouter_tache(request):
    """Traiter la création d'une nouvelle tâche (POST).

    Cette vue attend une requête POST contenant les champs du `TacheForm`.
    Si le formulaire est valide, la nouvelle tâche est sauvegardée et l'utilisateur
    est redirigé vers la page de liste des tâches.

    Args:
        request (django.http.HttpRequest): requête HTTP POST contenant les données du formulaire.

    Returns:
        django.http.HttpResponseRedirect: redirection vers la vue `accueil:liste`.
    """

    if request.method != 'POST':
        return redirect('accueil:liste')

    form = TacheForm(request.POST)
    if form.is_valid():
        form.save()
    return redirect('accueil:liste')


def saluer(request):
    """Traiter la soumission du formulaire de salutation (POST).

    Stocke le nom en session afin qu'il soit affiché ensuite par `liste_taches`.

    Args:
        request (django.http.HttpRequest): requête POST contenant le champ 'nom'.

    Returns:
        django.http.HttpResponseRedirect: redirection vers la vue `accueil:liste`.
    """

    if request.method == 'POST':
        form = NomForm(request.POST)
        if form.is_valid():
            request.session['nom'] = form.cleaned_data['nom']
    return redirect('accueil:liste')