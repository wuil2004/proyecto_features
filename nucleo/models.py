# nucleo/models.py
from django.db import models

class ReportePrecalculado(models.Model):
    # Esta es la "etiqueta" (1, 5, 10, 20...)
    n_features = models.IntegerField(primary_key=True, unique=True)

    # El JSON gigante con todos los resultados
    resultados_json = models.TextField()

    creado_en = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'reportes_precalculados_features'

    def __str__(self):
        return f"Reporte con {self.n_features} features"