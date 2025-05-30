from django import forms


class DatasetForm(forms.Form):
    name = forms.CharField(label="Dataset Name", max_length=50)
    #input_fields = forms.JSONField()
    #output_fields = forms.JSONField()
    csv_file = forms.FileField(allow_empty_file = False)

#class ParameterForm(forms.Form):
