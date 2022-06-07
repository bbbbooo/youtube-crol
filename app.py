from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def hello():
	return "This is the main page"

@app.route("/user",methods=['GET', 'POST'])
def post():
	if(request.method =='GET'):
		return render_template('index.html')

	elif(request.method == 'POST'):
		value = request.form['input']
		return render_template('result.html', name=value)

host_addr = '0.0.0.0'
port_num = '9090'

if __name__ == "__main__":
		app.run(host = host_addr, port = port_num)