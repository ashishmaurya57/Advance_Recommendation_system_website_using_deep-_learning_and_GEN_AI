# Generated by Django 5.1.8 on 2025-04-03 08:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("user", "0003_review"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="dob",
            field=models.DateField(blank=True, max_length=20, null=True),
        ),
    ]
