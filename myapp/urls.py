from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns= [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed')
]
 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)