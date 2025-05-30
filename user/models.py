from django.db import models

# Create your models here.
class contact(models.Model):
    name=models.CharField(max_length=100)
    email=models.CharField(max_length=120)
    mobile=models.CharField(max_length=20)
    message=models.CharField(max_length=600)
    def __str__(self):
        return self.email

class category(models.Model):
    cname=models.CharField(max_length=40)
    cpic=models.ImageField(upload_to='static/category/',default="")
    cdate=models.DateField()

    def __str__(self):
        return self.cname
class InterestTag(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
class profile(models.Model):
    name = models.CharField(max_length=120)
    dob = models.DateField(null=True, blank=True)
    mobile = models.CharField(max_length=20)
    email = models.EmailField(max_length=80, primary_key=True)
    passwd = models.CharField(max_length=100)
    ppic = models.ImageField(upload_to='static/profile/', default="")
    address = models.TextField(max_length=2000)
    purchased_products = models.ManyToManyField('product', related_name='buyers')
    
    # ✅ Replace textfield with tag-based M2M
    interests = models.ManyToManyField('InterestTag', blank=True)

    def __str__(self):
        return self.name

class SearchLog(models.Model):
    user = models.ForeignKey(profile, on_delete=models.CASCADE, related_name='search_logs')
    query = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.name} searched for '{self.query}'"
    

class login(models.Model):
    email=models.CharField(max_length=30,primary_key=True)
    passwd=models.CharField(max_length=20)
    
from .utils import compress_pdf
class product(models.Model):
    name = models.CharField(max_length=150)
    ppic = models.ImageField(upload_to='static/products', default="")
    language = models.CharField(max_length=40)
    hardcover = models.CharField(max_length=50)
    publisher = models.CharField(max_length=100)
    tprice = models.FloatField()
    disprice = models.FloatField()
    pdes = models.TextField(max_length=5000)
    category = models.ForeignKey(category, on_delete=models.CASCADE)
    pdate = models.DateField()
    pdf = models.FileField(upload_to='static/pdfs/', null=True, blank=True, help_text="Upload a PDF file for the product")
    likes = models.IntegerField(default=0)  # Total likes
    dislikes = models.IntegerField(default=0)  # Total dislikes
    tags = models.ManyToManyField('InterestTag', blank=True)


    def save(self, *args, **kwargs):
        # Compress the PDF before saving
        if self.pdf:
            self.pdf = compress_pdf(self.pdf)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
class order(models.Model):
    pid=models.IntegerField()
    userid=models.EmailField(max_length=100)
    remarks=models.CharField(max_length=40)
    status=models.BooleanField()
    odate=models.DateField()

class addtocart(models.Model):
    pid=models.IntegerField()
    userid=models.EmailField(max_length=100)
    status=models.BooleanField()
    cdate=models.DateField()

class Review(models.Model):
    product = models.ForeignKey(product, on_delete=models.CASCADE, related_name='reviews')  # Link to the product
    user = models.ForeignKey(profile, on_delete=models.CASCADE, related_name='reviews')  # Link to the user who reviewed
    rating = models.IntegerField(default=1, help_text="Rating between 1 and 5")  # Rating (1 to 5)
    comment = models.TextField(max_length=1000, blank=True, help_text="Optional review comment")  # Review comment
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the review was created

    def __str__(self):
        return f"{self.user.name} rated {self.product.name} ({self.rating}/5)"

class ProductInteraction(models.Model):
    user = models.ForeignKey(profile, on_delete=models.CASCADE, related_name='interactions')
    product = models.ForeignKey(product, on_delete=models.CASCADE, related_name='interactions')
    liked = models.BooleanField(default=False)  # True if liked, False if disliked
    disliked = models.BooleanField(default=False)  # True if disliked, False if liked
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        action = "liked" if self.liked else "disliked"
        return f"{self.user.name} {action} {self.product.name}"

# Example: user interactions model
from django.db import models
from django.utils import timezone

class UserInteraction(models.Model):
    INTERACTION_CHOICES = (
        ('view', 'View'),
        ('click', 'Click'),
        ('rating', 'Rating'),
        ('add_to_cart', 'Add to Cart'),
    )

    user = models.ForeignKey('profile', on_delete=models.CASCADE)
    product = models.ForeignKey('product', on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=50, choices=INTERACTION_CHOICES)
    rating = models.FloatField(null=True, blank=True)
    count = models.IntegerField(default=1)  # used for views/clicks
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('user', 'product', 'interaction_type')

    def __str__(self):
        return f"{self.user.email} - {self.product.name} - {self.interaction_type}"
