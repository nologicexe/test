from web_project import settings
from django.conf.urls.static import static
from django.urls import include
from AudioService.views import upload_file
from django.urls import path
from AudioService import views

urlpatterns = [
    path('AudioService/accounts/', include('django.contrib.auth.urls')),
    path("", views.home, name="home"),
    path("AudioService/UploadFile", views.upload_file, name="upload"),
    path("AudioService/<uploaded_filename>", views.results_redirected, name="uploaded_results"),
    path("AudioService/<filename>", views.results, name="results")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)