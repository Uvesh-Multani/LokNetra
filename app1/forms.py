# forms.py
from django import forms
# from .models import UserProfile

class UploadImageForm(forms.ModelForm):
    name = forms.CharField(label='Name', max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
    image = forms.ImageField(label='Image', widget=forms.FileInput(attrs={'class': 'form-control-file'}))

    # class Meta:
    #     model = UserProfile
    #     fields = ['phone_number', 'avatar']
