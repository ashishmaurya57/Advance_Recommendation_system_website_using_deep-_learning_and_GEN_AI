from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home_alt'),
    path('index/', views.home, name='index'),
    path('about/', views.about, name='about'),
    path('contactus/', views.contactus, name='contactus'),
    path('services/', views.services, name='services'),
    path('myorders/', views.myorders, name='myorders'),
    path('myprofile/', views.myprofile, name='myprofile'),
    path('products/', views.prod, name='products'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('viewdetails/', views.viewdetails, name='viewdetails'),
    path('process/', views.process, name='process'),
    path('logout/', views.logout, name='logout'),
    path('cart/', views.cart, name='cart'),
    path('search/', views.search_view, name='search_view'),
    path('handle_like_dislike/', views.handle_like_dislike, name='handle_like_dislike'),
    # path('api/recommendations/', views.get_recommendations_api, name='api_recommendations'),
    path('api/get_recommendations/', views.get_recommendations_api, name='get_recommendations_api'),
    path('set-language/', views.set_language, name='set_language'),
    path('api/translate-pdf-pages/', views.translate_pdf_pages_api, name='translate_pdf_pages_api'),
    path('start_payment/<int:pid>/', views.start_payment, name='start_payment'),
    path('payment_callback/', views.payment_callback, name='payment_callback'),
]