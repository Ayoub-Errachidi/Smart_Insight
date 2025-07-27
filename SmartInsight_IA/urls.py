from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]


from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("", include("projet_SI.urls")),
    path('admin/', admin.site.urls),
]
