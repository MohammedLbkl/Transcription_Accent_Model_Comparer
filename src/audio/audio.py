from IPython.display import HTML



def Ecoute_audio(file_path):
  return HTML(f"""
<audio controls>
  <source src="{file_path}" type="audio/mpeg">
  Votre navigateur ne supporte pas la balise audio.
</audio>
""") 