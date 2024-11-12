

from django.contrib import admin
from django.urls import path
from urloader import views  
urlpatterns = [
    path('admin/', admin.site.urls),
    path('url/', views.url, name="url"),  
    path('', views.index, name='index'),  
    path('summary/', views.summary, name='summary'),
]
