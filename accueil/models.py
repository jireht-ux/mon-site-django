from django.db import models


class Tache(models.Model):
	"""Modèle représentant une tâche simple.

	Champs:
	- title: titre de la tâche (max 200 caractères)
	- done: booléen indiquant si la tâche est terminée (False par défaut)
	"""

	title = models.CharField(max_length=200)
	done = models.BooleanField(default=False)

	def __str__(self) -> str:
		status = 'terminée' if self.done else 'en cours'
		return f"{self.title} ({status})"

