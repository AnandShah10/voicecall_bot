from django.db import models

# Create your models here.
class CompanyDocument(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    embedding = models.JSONField()  # stores embedding as list of floats

    def __str__(self):
        return self.title