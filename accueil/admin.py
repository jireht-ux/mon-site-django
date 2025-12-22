from django.contrib import admin
from .models import Tache


@admin.register(Tache)
class TacheAdmin(admin.ModelAdmin):
	list_display = ("title", "done")
	list_filter = ("done",)
	search_fields = ("title",)

