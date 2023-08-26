"""WebApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('run',views.run,name="run" ),
    path('',views.index,name="index" ),
    path('index.html',views.index,name="index" ),
    path('about.html',views.about,name="about" ),
    path('Services.html',views.services,name="services" ),
    path('contact.html',views.contact,name="contact" ),
    path('blog.html',views.blog,name="blog" ),
    path('Finance.html',views.finance,name="finance" ),
    path('Restaurant review.html',views.restaurant_review,name="Restaurant review" ),
    path('Restaurant review search.html',views.restaurant_review_search,name="Restaurant review search.html" ),
    path('api-auth/', include('rest_framework.urls')),
    path('api-search',views.res_rev_search,name="api"),
    path('api-search-finace',views.fin_search,name="fin_api")

]
