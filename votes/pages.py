

types = [
        'истероида',
        'эпилептоида',
        'паранойяла',
        'шизоида',
        'гипертима',
        'эмотива',
        'тревожного',
        'депрессивного',
        ]




def index():
    
    html = '''
    
<html>
    <head>
        <meta charset="utf-8">
        <title> Начать разметку </title>
    
    <body>
        <div class="form", align="center">
            <form action=/vote method="get">
                <p>
                    <label for="user_id">ID юзера</label>
                    <input type="text" id="user_id" name="user_id" length="8" value="" required=true>
                    <input type="submit" value="Начать">
                </p>
            </form>
    </body>
</html>
    '''

    return html




def votes(user_id, type_id, images, per_line=4):
    
    
    var = types[type_id]

    head = f'''

<html>
    <head>
        <meta charset="utf-8">
        <title> Разметка </title>

        <script type = "text/javascript">

            function select(id) {{

                var answer_id = document.getElementById('answer_'+id).value
                document.getElementById('answer_'+id).value = 1 - answer_id
                document.getElementById('div_'+id).style.opacity = .5 + answer_id * .5
                }}

        </script>

        <style>
            input {{
                
                font-family: Proxima Nova;
                font-size: 20px;
                line-height: normal;
                letter-spacing: 4px;
                
                background-color: #101010;
                color: grey;
                border: 0px;

                }}
                
         div_td {{
                 background-color: #101010;
                 color: grey;
                 border: 10px;
                }}
             
        </style>
    </head>

    <body>
        <h2 align="center">Выберите тех, у кого есть признаки <u>{var}</u> </h2>
        <div class="form">
            <form action=/vote method="post">
                <input type="hidden" id="user_id" name="user_id" value="{user_id}"/>
                <input type="hidden" id="type_id" name="type_id" value="{type_id}"/>
                
                <table>
                    <tr>
                        <td>
    '''

    cells = [f'''
                            <div class="div_td" id="div_{e}">
                                <input type="hidden" id="image_{e}" name="image_{e}" value="{image}"/>
                                <input type="hidden" id="answer_{e}" name="answer_{e}" value="0"/>
                                <img src="/images/{image}.jpg" height="320" onClick=select({e})></img>
                            </div>
            ''' for e, image in enumerate(images)]

    rows = [cells[c:c+per_line] for c in range(0,len(cells),per_line)]

    table = '''
                        </td>
                    </tr>
                    <tr>
                        <td>
    '''.join('''
                        </td>
                        <td>
    '''.join(row) for row in rows)

    
    foot = '''
                        </td>
                    </tr>
                </table>
                <p align="center">
                    <input type="submit" value=">          Дальше          <"/>
                </p>
            </form>
    </body>
</html>
    '''

    return head + table + foot




























 
