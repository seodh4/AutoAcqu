import pymysql
import os, datetime, json
import pandas as pd
from os.path import *

def insertDB_from_json(json_file, json_num, img_path, img_name):

    conn = pymysql.connect(host='127.0.0.1', user='AI', password='ai1234', db='Test', charset='utf8')
    cur = conn.cursor()

    json_data = json_file
    image_name = json_data["imagePath"]
    image_width = json_data["imageWidth"]
    image_height = json_data["imageHeight"]
    image_path = img_path+'/img/'
    table_name = image_path.split('/')[2] + '_image'
    label_name = image_path.split('/')[2] + '_label'
    image_size = getsize(image_path+img_name)
    image_type = image_name.split('.')[-1]
    image_form = ''
    json_num = json_num + 1
    json_file2 = 'IMT'
    image_form = json_file2
    label_num = 0
    label_id = 1
    image_num = 1

    tt2 = os.path.getctime(image_path + image_name)
    create_date = datetime.datetime.fromtimestamp(tt2)

    #table = f'''CREATE TABLE {table_name} (img_name VARCHAR(50), img_path VARCHAR(50), img_width INT(11), img_height INT(11), img_size INT(11), img_create_date DATETIME, img_type VARCHAR(50), img_form VARCHAR(50))'''
    #print(table)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (img_name VARCHAR(50) NOT NULL PRIMARY KEY, img_path VARCHAR(50), img_width INT(11), img_height INT(11), img_size INT(11), img_create_date DATETIME, img_type VARCHAR(50), img_form VARCHAR(50))")
    conn.commit()

    sql = f"INSERT INTO {table_name} (img_name, img_path, img_width, img_height, img_size, img_create_date, img_type, img_form) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (image_name, image_path, image_width, image_height, image_size, create_date, image_type, image_form)
    #print(image_name, image_path)

    cur.execute(sql, val)
    conn.commit()

    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_csv('./data/test2/DB/image.csv')
    

    for idx in range(len(json_data["shapes"])):
        label_num = label_num + 1
        label_parts = json_data["shapes"][idx]['label']
        label_polygon = json_data["shapes"][idx]['points']
        label_width = label_polygon[1][0] - label_polygon[0][0]
        label_height = label_polygon[1][1] - label_polygon[0][1]
        label_area = label_width * label_height
        label_coordinates = str(label_polygon[0][0]) + '.' + str(label_polygon[0][1]) + '.' + str(label_polygon[1][0]) + '.' + str(label_polygon[1][1])
        #label_id = json_file2 + '_' + image_path.split('/')[2] + '_' + str(label_num)
        #print(label_id)

        #print(f'''CREATE TABLE IF NOT EXISTS {label_name}, (label_parts CHAR(3), label_width INT(11), label_height INT(11), label_area INT(11), label_coordinates VARCHAR(50), FOREIGN KEY(label_parts) REFERENCES {table_name}(img_name))''')
        #cur.execute(f'''CREATE TABLE IF NOT EXISTS {label_name} (label_parts VARCHAR(3) NOT NULL, label_width INT(11), label_height INT(11), label_area INT(11), label_coordinates VARCHAR(50), FOREIGN KEY(label_parts) REFERENCES {table_name} (img_id))''')
        #cur.execute(f"CREATE TABLE IF NOT EXISTS {label_name} (label_id VARCHAR(50), label_parts VARCHAR(50), label_width INT(11), label_height INT(11), label_area INT(11), label_coordinates VARCHAR(50))")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {label_name} (label_parts VARCHAR(50), label_width INT(11), label_height INT(11), label_area INT(11), label_coordinates VARCHAR(50), img_name VARCHAR(50) NOT NULL)")
        conn.commit()

        #sql2 = "INSERT INTO pcb_imt_label (label_parts, label_width, label_height, label_area, label_coordinates) VALUES (%s, %s, %s, %s, %s)"
        sql2 = f"INSERT INTO {label_name} (label_parts, label_width, label_height, label_area, label_coordinates, img_name) VALUES (%s, %s, %s, %s, %s, %s)"
        val2 = (label_parts, label_width, label_height, label_area, label_coordinates, image_name)

        #sql2 = f"INSERT INTO {label_name} (label_id, label_parts, label_width, label_height, label_area, label_coordinates) VALUES (%s, %s, %s, %s, %s, %s)"
        #val2 = (label_id, label_parts, label_width, label_height, label_area, label_coordinates)

        cur.execute(sql2, val2)
        conn.commit()

        df = pd.read_sql_query(f"SELECT * FROM {label_name}", conn)
        df.to_csv('./data/test2/DB/label.csv')

        label_id += 1

        #sql3 = f"ALTER TABLE {label_name} ADD FOREIGN KEY (label_parts) VALUES (%s) REFERENCES {table_name}(img_name) VALUES (%s)"
        #val3 = (label_parts, image_name)

        #cur.execute(sql3, val3)
        cur.execute(f"ALTER TABLE {label_name} ADD FOREIGN KEY (img_name) REFERENCES {table_name}(img_name)")
        #conn.commit()
    #label_id = 1
    #update_from_folder(json_file, json_num, json_file2, cur, conn, img_path)


def update_from_folder(json_file, json_num, check_folder_form, cur, conn, img_path):

    JSON_FOLDER = img_path #"\\".join(json_file.split('\\')[:-1])
    t = os.path.getmtime(JSON_FOLDER)
    modify_date = datetime.datetime.fromtimestamp(t)
    t2 = os.path.getctime(JSON_FOLDER)
    create_date = datetime.datetime.fromtimestamp(t2)
    category = ''
    
    if "IMT" in check_folder_form:
        category = 'PCB'

    sql3 = "INSERT INTO pcb_imt_folder (folder_category, image_count, folder_path, create_date, modify_date, version) VALUES (%s, %s, %s, %s, %s, %s)"
    val3 = (category, json_num, JSON_FOLDER, create_date, modify_date, 1.0)

    print(category, json_num, JSON_FOLDER, create_date, modify_date, 1.0)

    cur.execute(sql3, val3)
    conn.commit()







