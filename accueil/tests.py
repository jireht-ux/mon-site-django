from django.test import TestCase
from .models import Tache


class TacheModelTest(TestCase):
	def test_str_returns_title(self):
		"""La méthode __str__ doit contenir le titre de la tâche."""
		titre = "Tester la méthode __str__"
		t = Tache.objects.create(title=titre, done=False)
		self.assertIn(titre, str(t))

