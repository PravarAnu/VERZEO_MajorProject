from django.urls import path

from skintone import views

app_name = 'skin'
urlpatterns = [
    path('', views.home, name='home'),
    path('coding/', views.coding, name='coding'),
    path('readmore/', views.readmore, name='readmore'),
    path('robust/', views.robust, name='robust'),
    path('thankyou/', views.thankyou, name='thankyou'),
    path('preview/', views.preview, name='preview'),
    path('imgprocess/', views.imgprocess, name='imgprocess')
]
