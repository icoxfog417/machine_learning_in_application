var MAIN_ELEMENT = "#main"
var CANVAS_ID = "canvas"

var Prediction = (function () {
    function Prediction(image, sample) {
        this.image = image;
        this.sampleImage = sample[0];
        this.sampleData = sample[1];
        this.result = -1;
    }

    Prediction.prototype.envelop = function (data) {
        var getCookie = function(name){
            var r = document.cookie.match("\\b" + name + "=([^;]*)\\b");
            return r ? r[1] : undefined;
        }
        var envelope = {
            _xsrf: getCookie("_xsrf"),
            "data[]": data
        }
        return envelope;
    }

    Prediction.prototype.imageSrc = function () {
        return this.image.toDataURL();
    }

    Prediction.prototype.execute = function () {
        var self = this;
        var d = new $.Deferred;
        $.post("/predict", self.envelop(self.sampleData), function(prediction){
            self.result = prediction["result"];
            d.resolve(self)
        })
        return d.promise();
    };

    Prediction.prototype.feedback = function (value) {
        var self = this;
        var d = new $.Deferred;
        var feedback = [parseInt(value)];
        feedback = feedback.concat(self.sampleData);
        $.post("/feedback", self.envelop(feedback), function(feedbacked){
            if(feedbacked["result"] == ""){
                self.result = feedback[0];
                d.resolve();
            }else{
                d.reject(feedbacked["result"]);
            }
        })
        return d.promise();
    };

    return Prediction;
})();

Vue.config.delimiters = ["[[", "]]"];
Vue.config.prefix =  "data-v-";
Vue.component("predict-item", {
    template: "#predict-item",
    methods: {
        beginEdit: function(){
            this.state.editing = true;
        },
        endEdit: function(){
            var state = this.state;
            if(state.value >= 0 && state.value < 10 && (state.value != this.result)){
                var original = this.result;
                this.$data.feedback(state.value).fail(function(msg){
                    state.value = original;
                })
            }else{
                state.value = this.result;
            }
            state.editing = false;
        }
    }
});
var app = new Vue({
    el: MAIN_ELEMENT,
    data: {
        canvas: null,
        SNAP_SIZE: 120,
        SAMPLE_SIZE: 80,
        predicts: []
    },
    created: function(){
        this.canvas = new Canvas(CANVAS_ID, {
            strokeStyle: "black"
        });
    },
    methods:{
        clear: function(){
            this.canvas.clear();
        },
        injectState: function(p){
            p.state =  {
                editing: false,
                value: p.result
            }
        },
        submit: function(){
            var self = this;
            var image = self.canvas.snapShot(self.SNAP_SIZE);
            var sample = self.canvas.toSample(self.SAMPLE_SIZE, self.SAMPLE_SIZE);
            var total = sample[1].reduce(function(a, b){ return a + b; });
            if(total == 0){
                return false;
            }
            var p = new Prediction(image, sample);
            p.execute().done(function(p){
                self.injectState(p);
                self.predicts.unshift(p);
                self.clear();
            })
        }
    }
});
