from django.shortcuts import render

# Create your views here.
# nucleo/views.py

from django.http import HttpResponse
from django.shortcuts import render
from .models import ReportePrecalculado # El modelo
import json

def home_page(request):
    """Muestra la página de inicio con el dropdown."""
    # Busca en la DB qué reportes ya están cocinados
    reportes = ReportePrecalculado.objects.all().order_by('n_features')

    # Obtenemos la lista de N features disponibles (ej: [1, 5, 10, 20, 30...])
    features_disponibles = [r.n_features for r in reportes]

    context = {
        'features_disponibles': features_disponibles
    }
    # Aún no creamos el HTML, pero la función ya existe
    return render(request, 'nucleo/index.html', context) 

def ver_reporte(request, n_features):
    """Busca un reporte pre-calculado y lo muestra."""
    try:
        job = ReportePrecalculado.objects.get(n_features=n_features)

        # Carga el JSON guardado y lo pasa al template
        context = json.loads(job.resultados_json) 

        # Añadimos formato a los números para que se vean bien
        context['f1_score_base_pct'] = f"{context['f1_score_base'] * 100:.2f}%"
        context['f1_score_reduced_pct'] = f"{context['f1_score_reduced'] * 100:.2f}%"
        context['f1_score_difference_pct'] = f"{context['f1_score_difference'] * 100:.2f}%"

        # Aún no creamos el HTML, pero la función ya existe
        return render(request, 'nucleo/resultado.html', context) 

    except ReportePrecalculado.DoesNotExist:
        return HttpResponse(
            f"<h1>Reporte para {n_features} features no encontrado.</h1>"
            f"<p>Asegúrate de haber corrido 'python manage.py generar_reportes_features {n_features}' primero.</p>",
            status=404
        )