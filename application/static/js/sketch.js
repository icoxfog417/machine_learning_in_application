var Dot = (function () {
    function Dot(x, y) {
        this.x = x;
        this.y = y;
    }
    return Dot;
})();

var Stroke = (function () {
    function Stroke() {
        this.dots = [];
    }
    Stroke.prototype.draw = function (dot) {
        this.dots.push(dot);
    };
    return Stroke;
})();

var Canvas = (function () {
    function Canvas(canvasId, pencil) {
        var self = this
        self.$canvas = $("#" + canvasId);
        self.drawing = false;
        self.strokes = [];

        self.pencil = {
            strokeStyle: "#df4b26",
            lineJoin: "round",
            lineWidth: 10
        };

        if(arguments.length > 1){
            for(var k in self.pencil){
                if(k in pencil){
                    self.pencil[k] = pencil[k];
                }
            }
        }

        self.$canvas
        .on("mousedown", function(e){
            var d = self.getPosition(e);
            self.draw(d);
        })
        .on("mousemove", function(e){
            if(self.drawing){
                var d = self.getPosition(e);
                self.draw(d);
            }
        })
        .on("mouseup mouseleave", function(e){
            self.drawing = false;
        })

    }

    Canvas.prototype.getCanvas = function () {
         return this.$canvas.get(0);
    }

    Canvas.prototype.getContext = function () {
         return this.getCanvas().getContext("2d");
    }

    Canvas.prototype.getPosition = function (event) {
        var canvasOffset = this.$canvas.offset();
        var relX = event.pageX - canvasOffset.left;
        var relY = event.pageY - canvasOffset.top;
        return new Dot(relX, relY);
    }

    Canvas.prototype.draw = function (dot) {
        var stroking = null;
        if(!this.drawing){
            stroking = new Stroke();
            this.strokes.push(stroking);
            this.drawing = true;
        }else{
            stroking = this.strokes[this.strokes.length - 1];
        }

        if(stroking != null){
            stroking.draw(dot);
            this.flush();
        }
    };

    Canvas.prototype.flush = function(){
        var context = this.getContext();
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);

        context.strokeStyle = this.pencil.strokeStyle;
        context.lineJoin = this.pencil.lineJoin;
        context.lineWidth = this.pencil.lineWidth;

        for(var i = 0; i < this.strokes.length; i++) {
            var s = this.strokes[i];
            var preDot = null;
            for(var j = 0; j < s.dots.length; j++){
                context.beginPath();

                var d = s.dots[j];
                if(preDot == null){
                    context.moveTo(d.x, d.y);
                }else{
                    context.moveTo(preDot.x, preDot.y);
                }
                context.lineTo(d.x, d.y);
                preDot = d;

                context.closePath();
                context.stroke();
            }
        }
    }

    Canvas.prototype.clear = function(){
        this.strokes = [];
        this.drawing = false;

        var context = this.getContext();
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    }

    Canvas.prototype.snapShot = function(x, y){
        var snap = document.createElement("canvas");

        if(arguments.length < 1){
            snap.width = this.getCanvas().width;
            snap.height = this.getCanvas().height;
            var context = snap.getContext("2d");
            context.drawImage(this.getContext().canvas, 0, 0);
        }else if(arguments.length < 2){
            snap.width = snap.height = x;
            var context = snap.getContext("2d");
            context.drawImage(this.getContext().canvas, 0, 0, x, x);
        }else{
            snap.width = x;
            snap.height = y;
            var context = snap.getContext("2d");
            context.drawImage(this.getContext().canvas, 0, 0, x, y);
        }

        return snap;
    }

    Canvas.prototype.toSample = function(x, y){
        var sample = this.snapShot(x, y);
        var ctx = sample.getContext("2d");

        var src = ctx.getImageData(0, 0, x, y);
        var dst = ctx.createImageData(x, y);
        var data = [];
        for (var i = 0; i < src.data.length; i += 4) {
            var rgb = src.data[i] + src.data[i+1] + src.data[i+2];
            var sum = rgb + src.data[i+3];
            data.push(Math.sqrt(Math.min(sum,255)))
            dst.data[i] = dst.data[i+1] = dst.data[i+2] = rgb / 3;
            dst.data[i+3] = src.data[i+3];
        }

        ctx.putImageData(dst, 0, 0);
        return [sample, data]

    }

    return Canvas;
})();