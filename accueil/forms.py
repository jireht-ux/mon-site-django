from django import forms

class NomForm(forms.Form):
    nom = forms.CharField(label="Votre nom", max_length=100)