$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
				var res = "";
				if(data == "africangrey")
					res = "Сірий Африканський (Жако)";
				if(data == "budgerigar")
					res = "Хвилястий папужка";
				if(data == "cockatiel")
					res = "Корелла";
				if(data == "cockatoo")
					res = "Какаду";
				if(data == "conure")
					res = "Сонячний папужка";
				if(data == "lovebird")
					res = "Нерозлучник";
				if(data == "macaw")
					res = "Гіацинтовий Ара";
				if(data == "random" || data =="")
					res = "зовсім не папуга";
                $('.loader').hide();
                $('#result').fadeIn(600);
				if(!res.length) res  = "зовсім не папуга";
                $('#result').text(' Схоже, що це: ' + res);
                console.log('Success!'+data+"___"+res);
            },
        });
    });

});
