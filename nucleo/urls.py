# nucleo/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('reporte/<int:n_features>/', views.ver_reporte, name='ver_reporte'),
]