"""
College & University 19756
Residence 4845
Arts & Entertainment 32414
Food 546359
Shop & Service 293435
Nightlife Spot 100810
Travel & Transport 103169
Outdoors & Recreation 73141
Professional & Other Places 122756
Event 0
"""

#Can consider using editdistance instead of handcraft
errata = {'Ramen /  Noodle House': 'Noodle House',
          'Drugstore / Pharmacy' : 'Drugstore',
          'Mall' : 'Shopping Mall',
          'Subway': 'Metro Station',
          'Spa / Massage': 'Spa', 
          'Gas Station / Garage' : 'Gas Station',
          'Athletic & Sport' : 'Athletics & Sports',
          'General College & Univers': 'General College & University',
          'Light Rail' : 'Light Rail Station',
          'Paper / Office Supplies S' : 'Paper / Office Supplies Store',
          'Residential Building (Apa':'Residential Building (Apartment / Condo)',
          'Car Dealership':'Auto Dealership',
          'Ferry':'Boat or Ferry',
          'Vegetarian / Vegan Restau': 'Vegetarian / Vegan Restaurant',
          'Eastern European Restaura': 'Eastern European Restaurant', 
          'Financial or Legal Servic' : 'Financial or Legal Service',
          'Malaysian Restaurant': 'Malay Restaurant',
          'Southern / Soul Food Rest':'Southern / Soul Food Restaurant',
          'Professional & Other Plac': 'Professional & Other Places'}

def correct_errata(name):
    if name in errata:
        return errata[name]
    else:
        return name