from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class UserRegisterForm(UserCreationForm):
    username = forms.CharField(label="Username", widget=forms.TextInput(
        attrs={'class': 'form-control'})
    )
    email = forms.EmailField(label="Email", widget=forms.EmailInput(
        attrs={'class': 'form-control'})
    )
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput(
        attrs={'class' : 'form-control'})
    )
    password2 = forms.CharField(label="Confirm Password", widget=forms.PasswordInput(
        attrs={'class' : 'form-control'})
    )

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_active = False
        if commit:
            user.save()
        return user

class UserLoginForm(AuthenticationForm):
    username = forms.CharField(label="Username", widget=forms.TextInput(
        attrs={'class': 'form-control'})
    )
    password = forms.CharField(label="Password", widget=forms.PasswordInput(
        attrs={'class' : 'form-control'})
    )