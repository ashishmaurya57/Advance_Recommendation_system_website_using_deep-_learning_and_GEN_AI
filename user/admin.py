from django.contrib import admin
from django.utils.html import format_html
from .models import *

class contactAdmin(admin.ModelAdmin):
    list_display = ("name", "mobile", "email", "message")
admin.site.register(contact, contactAdmin)

class categoryAdmin(admin.ModelAdmin):
    list_display = ("id", "cname", "cpic", "cdate")
admin.site.register(category, categoryAdmin)

@admin.register(InterestTag)
class InterestTagAdmin(admin.ModelAdmin):
    list_display = ("name",)

class profileAdmin(admin.ModelAdmin):
    list_display = ("name", "dob", "mobile", "email", "passwd", "ppic", "address", "display_interests")
    filter_horizontal = ("interests",)  # Allows tag-like multi-select in admin UI

    def display_interests(self, obj):
        return ", ".join([tag.name for tag in obj.interests.all()])
    display_interests.short_description = "Interests"
admin.site.register(profile, profileAdmin)

class productAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "ppic", "language", "hardcover", "publisher", "tprice", "disprice", "pdes", "category", "pdate", "pdf_link")
    list_filter = ("category", "pdate")
    search_fields = ("name", "category__cname", "language", "publisher")

    def pdf_link(self, obj):
        if obj.pdf:
            return format_html('<a href="{}" target="_blank">View PDF</a>', obj.pdf.url)
        return "No PDF"
    pdf_link.short_description = "PDF File"
admin.site.register(product, productAdmin)

class orderAdmin(admin.ModelAdmin):
    list_display = ("id", "pid", "userid", "remarks", "status", "odate")
admin.site.register(order, orderAdmin)

class addtocartAdmin(admin.ModelAdmin):
    list_display = ("id", "pid", "userid", "status", "cdate")
admin.site.register(addtocart, addtocartAdmin)

class ReviewAdmin(admin.ModelAdmin):
    list_display = ('product', 'user', 'rating', 'created_at')
    search_fields = ('product__name', 'user__name', 'rating')
admin.site.register(Review, ReviewAdmin)

@admin.register(UserInteraction)
class UserInteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'interaction_type', 'rating', 'count', 'timestamp')
    list_filter = ('interaction_type', 'timestamp')
    search_fields = ('user__name', 'product__name')

    def get_rating(self, obj):
        return obj.rating if obj.rating is not None else '-'
    get_rating.short_description = 'Rating'
    get_rating.admin_order_field = 'rating'



