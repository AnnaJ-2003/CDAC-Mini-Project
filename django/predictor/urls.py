from django.urls import path
from . import views

urlpatterns = [
    path("", views.analysis, name="analysis"),
    path("predict/", views.predict, name="predict"),
]