from django.urls import path
from Engine import views

urlpatterns = [
    path('Predict/',views.predict_image,name='Predict'),
    
]
