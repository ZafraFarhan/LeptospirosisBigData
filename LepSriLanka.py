
!pip install requests beautifulsoup4 pdfplumber


import requests
from bs4 import BeautifulSoup
import os

# Define the URL of the webpage
url = 'https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report'

# Define the directory to save downloaded PDFs
output_dir = 'pdfs'
os.makedirs(output_dir, exist_ok=True)

def download_first_pdf(url, output_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first accordion
    accordion = soup.find(class_='accordions')

    downloaded_pdfs = []

    if accordion:
        # Find the content within the first accordion
        content = accordion.find(class_='content')
        if content:
            # Find the first product link within the content
            product = content.find('li', class_='product')
            if product:
                a_tag = product.find('a', href=True)
                if a_tag:
                    pdf_url = a_tag['href']
                    pdf_name = os.path.basename(pdf_url)
                    pdf_path = os.path.join(output_dir, pdf_name)

                    # Download the PDF
                    pdf_response = requests.get(pdf_url)
                    with open(pdf_path, 'wb') as file:
                        file.write(pdf_response.content)

                    print(f'Downloaded: {pdf_path}')
                    downloaded_pdfs.append(pdf_path)

    return downloaded_pdfs

pdf_paths1 = download_first_pdf(url, output_dir)


import requests
from bs4 import BeautifulSoup
import os

# Define the URL of the webpage
url = 'https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report'

# Define the directory to save downloaded PDFs
output_dir = 'pdfs'
os.makedirs(output_dir, exist_ok=True)

def download_last_pdf(url, output_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all accordions
    accordions = soup.find_all(class_='accordions')

    downloaded_pdfs = []

    for accordion in accordions:
        # Find all content within the accordion
        content = accordion.find(class_='content')
        if content:
            # Find all product links within the content
            products = content.find_all('li', class_='product')
            if products:
                # Get the last product link
                last_product = products[-1]
                a_tag = last_product.find('a', href=True)
                if a_tag:
                    pdf_url = a_tag['href']
                    pdf_name = os.path.basename(pdf_url)
                    pdf_path = os.path.join(output_dir, pdf_name)

                    # Download the PDF
                    pdf_response = requests.get(pdf_url)
                    with open(pdf_path, 'wb') as file:
                        file.write(pdf_response.content)

                    print(f'Downloaded: {pdf_path}')
                    downloaded_pdfs.append(pdf_path)

    return downloaded_pdfs

pdf_paths2 = download_last_pdf(url, output_dir)


pdf_paths = pdf_paths1 + pdf_paths2
print(pdf_paths)

pdf_paths.pop()
print(pdf_paths)


table_title = 'Selected notifiable diseases reported by Medical Officers of Health'


unrotated_pdfs = pdf_paths[12:19][::-1]
unrotated_pdfs



import requests
from bs4 import BeautifulSoup
import os
import pdfplumber
import pandas as pd


def extract_table_from_pdf(pdf_path,table_title):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if table_title in text:
                # Extract tables
                tables = page.extract_tables()
                print('table found')

                 # Initialize an empty DataFrame to hold combined data
                combined_data = pd.DataFrame()

                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])

                    # Check for column existence and set up the DataFrame appropriately
                    if 'DPDHS\nDivision' in df.columns:
                        df = df.iloc[1:27, 12]  # Skip the first row, which might be a header
                    else:
                        df = df.iloc[3:29, 12]  # Skip the first few rows if no 'DPDHS\nDivision' column

                    df.reset_index(drop=True, inplace=True)

                    #combined_data.rename(columns={df.columns[0]: 'Total'}, inplace=True)

                    # Add data to the combined_data DataFrame
                    combined_data =  pd.concat([combined_data, df], ignore_index=True, sort=False)



                return combined_data  # Return the combined data

    return None  # Return None if no matching table is foun

# Initialize an empty DataFrame to store all combined data
all_combined_data = pd.DataFrame()

# Iterate over the list of PDF paths
for pdf_path in unrotated_pdfs:
    extracted_data = extract_table_from_pdf(pdf_path, table_title)
    if extracted_data is not None:
        all_combined_data = pd.concat([all_combined_data, extracted_data], ignore_index=True, sort=False)

        print(f"Extracted data from {pdf_path}:")
        print(extracted_data)
    else:
        print(f"No matching table found in {pdf_path}")



all_combined_data.head()

all_combined_data.columns = ['Cases']

import pandas as pd

# Function to split rows with newline characters and expand them
def split_and_expand(df, col1):
    # Handle missing values by filling them with empty strings
    df[col1] = df[col1].fillna('None')  # Fill NaN values with empty strings

    # Filter rows that contain newline characters
    mask = df[col1].str.contains('\n')

    # Split and expand only the rows with newline characters
    expanded_rows = df[mask].apply(lambda x: pd.Series(x[col1].split('\n')), axis=1).stack().reset_index(level=1, drop=True).rename(col1)

    # Combine the expanded data into a new DataFrame
    expanded_df = pd.DataFrame({col1: expanded_rows}).reset_index(drop=True)

    # Append the rows that did not need expansion
    non_expanded_df = df[~mask].reset_index(drop=True)

    # Concatenate expanded and non-expanded DataFrames
    cleaned_df = pd.concat([expanded_df, non_expanded_df], ignore_index=True)

    return cleaned_df

# Clean the dataset
unrotated_combined_data = split_and_expand(all_combined_data.copy(), 'Cases')  # Create a copy to avoid modifying the original

# Display the cleaned DataFrame
print(unrotated_combined_data)


# Drop rows 26 and 52
unrotated_combined_data = unrotated_combined_data.drop([52, 53])

# Reset the index after dropping rows
unrotated_combined_data = unrotated_combined_data.reset_index(drop=True)

# Display the modified DataFrame
print(unrotated_combined_data)


import pandas as pd

def split_into_columns(df, num_rows_per_column):
    # Calculate the number of columns needed
    num_columns = (len(df) + num_rows_per_column - 1) // num_rows_per_column

    # Initialize a list to hold the columns
    columns = []

    for i in range(num_columns):
        # Slice the DataFrame to get the desired rows for this column
        column_data = unrotated_combined_data.iloc[i*num_rows_per_column:(i+1)*num_rows_per_column].reset_index(drop=True)

        # Append the series as a new column
        columns.append(column_data)

    # Concatenate the columns to form the final DataFrame
    final_df = pd.concat(columns, axis=1)

    return final_df


# Split the DataFrame into columns with 25 rows each
unrotated_final_df = split_into_columns(unrotated_combined_data['Cases'], 26)



# Print the final DataFrame
print(unrotated_final_df)




!pip install pdf2image pillow pdfplumber
!apt-get install poppler-utils


from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
import pandas as pd

def rotate_image(image_path, angle):
    with Image.open(image_path) as img:
        rotated_img = img.rotate(angle, expand=True)
        rotated_img.save('rotated_image.png')

def extract_table_from_pdf(pdf_path, table_title):
    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    images[0].save('page_image.png')

    # Rotate the image if needed (e.g., 90 degrees)
    rotate_image('page_image.png', 90)

    # Use pdfplumber on the rotated image
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if table_title in text:
                tables = page.extract_tables()
                print('Table found')

                # Initialize an empty DataFrame to hold combined data
                combined_data = pd.DataFrame()

                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df = df.applymap(lambda x: ''.join(x.split())[::-1] if isinstance(x, str) else x)

                    # Check for column existence and set up the DataFrame appropriately
                    if pdf_path=='pdfs/en_66b1cf8a4ad9f_Vol_51_no_29-english.pdf':
                        df = df.iloc[15, 2:28]  # Skip the first row, which might be a header
                    else:
                        df = df.iloc[13, 2:28]  # Skip the first few rows if no 'DPDHS\nDivision' column

                    df.reset_index(drop=True, inplace=True)

                    # Add data to the combined_data DataFrame
                    combined_data = pd.concat([combined_data, df], ignore_index=True, sort=False)

                return combined_data  # Return the combined data

    return None  # Return None if no matching table is found



rotated_paths = pdf_paths[0:12][::-1]
rotated_paths

# Initialize an empty DataFrame to store all combined data
rotated_combined_data = pd.DataFrame()

# Iterate over the list of PDF paths
for pdf_path in rotated_paths:
    extracted_data = extract_table_from_pdf(pdf_path, table_title)
    if extracted_data is not None:
        rotated_combined_data = pd.concat([rotated_combined_data, extracted_data], ignore_index=True, sort=False)

        print(f"Extracted data from {pdf_path}:")
        print(extracted_data)
    else:
        print(f"No matching table found in {pdf_path}")

# Output the combined data from all PDFs
print("Rotated combined data:")
print(rotated_combined_data)

rotated_combined_data.columns = ['Cases']

# Clean the dataset
rotated_combined_data = split_and_expand(rotated_combined_data.copy(), 'Cases')  # Create a copy to avoid modifying the original

# Display the cleaned DataFrame
print(rotated_combined_data)

rotated_combined_data.head()



import pandas as pd

def split_into_columns(df, num_rows_per_column):
    # Calculate the number of columns needed
    num_columns = (len(df) + num_rows_per_column - 1) // num_rows_per_column

    # Initialize a list to hold the columns
    columns = []

    for i in range(num_columns):
        # Slice the DataFrame to get the desired rows for this column
        column_data = rotated_combined_data.iloc[i*num_rows_per_column:(i+1)*num_rows_per_column].reset_index(drop=True)

        # Append the series as a new column
        columns.append(column_data)

    # Concatenate the columns to form the final DataFrame
    final_df = pd.concat(columns, axis=1)

    return final_df


# Split the DataFrame into columns with 25 rows each
rotated_final_df = split_into_columns(rotated_combined_data['Cases'], 26)


# Print the final DataFrame
print(rotated_final_df)




complete_df=unrotated_final_df.merge(rotated_final_df, left_index=True, right_index=True)
print(complete_df)

Countries = ['Colombo','Gampaha','Kalutara','Kandy','Matale','Nuwara Eliya','Galle','Hambantota','Matara','Jaffna','Kilinochchi','Mannar','Vavuniya','Mullaitivu','Batticaloa','Ampara','Trincomalee',
             'Kurunegala','Puttalam','Anuradhapura','Polonnaruwa','Badulla','Monaragala','Ratnapura','Kegalle','Kalmunai']

complete_df.index = Countries
complete_df.head()

complete_df.columns = ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
complete_df.head()



# Reshape the DataFrame from wide to long format
df_long = pd.melt(complete_df.reset_index(), id_vars=['index'], var_name='Year', value_name='Cases')

# Rename the 'index' column to 'Country'
df_long = df_long.rename(columns={'index': 'Country'})

# Sort the DataFrame first by Year, then by Country
df_long = df_long.sort_values(by=['Year', 'Country']).reset_index(drop=True)

# Display the transformed DataFrame
print(df_long)

df_long.to_csv('Leptospirosis_SriLanka.csv', index=False)  # Save DataFrame to CSV

# Download the CSV file
from google.colab import files
files.download('Leptospirosis_SriLanka.csv')
