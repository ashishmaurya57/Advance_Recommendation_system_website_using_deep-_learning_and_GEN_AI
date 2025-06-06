# Generated by Django 5.1.8 on 2025-06-01 12:57

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("user", "0012_product_disliked_users_product_liked_users_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="userinteraction",
            name="interaction_type",
            field=models.CharField(
                choices=[
                    ("view", "View"),
                    ("click", "Click"),
                    ("rating", "Rating"),
                    ("add_to_cart", "Add to Cart"),
                    ("like", "Like"),
                    ("dislike", "Dislike"),
                ],
                max_length=50,
            ),
        ),
        migrations.CreateModel(
            name="UserInterest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("timestamp", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "interest_tag",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="user.interesttag",
                    ),
                ),
                (
                    "user_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="user.profile"
                    ),
                ),
            ],
            options={
                "unique_together": {("user_profile", "interest_tag")},
            },
        ),
        migrations.AlterField(
            model_name="profile",
            name="interests",
            field=models.ManyToManyField(
                blank=True, through="user.UserInterest", to="user.interesttag"
            ),
        ),
    ]
