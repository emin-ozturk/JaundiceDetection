var model, jaundice, normal, image, elem1, elem2, text1, text2, pPredict;

//secilen goruntu uzerindeislemler yaparak 
//sarilik olup olmadigi kontrol edilir
const detection = async () => {
	const offset = tf.scalar(127.5);	
	var input = tf.browser.fromPixels(image).toFloat()
	input = input.sub(offset).div(offset);
	input = input.resizeBilinear([224,224]).reshape([1,224,224,3])
	result = model.predict(input).dataSync()
	result = Array.from(result)
	jaundice = (result[1]*100).toFixed(2).toString() + "%";
	normal = (result[0]*100).toFixed(2).toString() + "%";
			
	text1.innerHTML = "Sarılık: " + jaundice;
	text2.innerHTML = "Normal: " + normal;
	elem1.style.width = jaundice;
	elem2.style.width = normal;
	
	var predictRatio, predict;
	if (result[0] < result[1]) {
		predictRatio = jaundice;
		predict = "Sarılık"
	} else {
		predictRatio = normal;
		predict = "Normal"
	}
	
	pPredict.innerHTML = "Seçilen görüntü " + predictRatio + " oranında " + predict + " olarak tahmin edilmiştir.";
			
};
const init = async () => {
	//ilk acilista model dosyasını yukler
	model = await tf.loadLayersModel('../static/model/model.json');
	//degiskenlerin alacagi degerler
	text1 = document.getElementById("text1");
	text2 = document.getElementById("text2");
	elem1 = document.getElementById("progressBar1");
	elem2 = document.getElementById("progressBar2");
	pPredict = document.getElementById("pPredict");
};

$('#btnImageSelect').on('change', function() {
	$('#imgPreview').attr('src', window.URL.createObjectURL(this.files[0]))
	$('#imgPreview').on('load', function() {
		image = document.getElementById('imgPreview')
			
		$('#imgCanvas').css('border', 'none')
		$('#imgCanvas').attr('width', image.width)
		$('#imgCanvas').attr('height', image.height)
			
		var canvas = document.getElementById('imgCanvas')
		var ctx = canvas.getContext('2d')
			
		ctx.drawImage(image, 0, 0, image.width, image.height)
		detection();
	})
})

init();
