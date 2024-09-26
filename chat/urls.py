from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # URL for input form
    path('get_response/', views.get_response, name='get_response'),  # URL for processing the input
]
