from django.db import models


class Tache(models.Model):
	"""Modèle représentant une tâche simple.

	Champs:
	- title: titre de la tâche (max 200 caractères)
	- done: booléen indiquant si la tâche est terminée (False par défaut)
	"""

	title = models.CharField(max_length=200)
	done = models.BooleanField(default=False)

	class Meta:
		verbose_name = "Tâche"
		verbose_name_plural = "Tâches"

	def __str__(self):
		return f"{self.title} ({'terminée' if self.done else 'en cours'})"

