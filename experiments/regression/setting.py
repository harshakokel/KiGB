data_target ={"abalone":"Rings",
              "autompg":"mpg",
              "autoprice":"price",
              "boston":"MEDV",
              "california":"medianHouseValue",
              "cpu":"logPRP",
              "crime":"ViolentCrimesPerPop",
              "redwine":"quality",
              "whitewine":"quality",
              "windsor":"sell",
              "turvo":"shipment_price"
              }

data_advice={"abalone":"+1,0,0,0,0,0,+1,0,0,0",
             "autompg":"-1,-1,-1,-1,+1,+1,+1",
             "autoprice":"0,0,0,0,0,0,0,0,+1,+1,+1,+1,+1",
              "boston":"-1,0,0,0,+1,0,0,0,0,-1,0,0",
              "california":"0,+1,+1,0,0,0",
              "cpu":"0,+1,+1,+1,0,0",
              "crime":"+1,0,+1,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,+1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
              "redwine":"0,-1,+1,0,0,0,0,0,0,+1,+1",
              "whitewine":"0,-1,+1,0,0,0,0,0,0,+1,+1",
             "windsor":"+1,0,0,0,0,0,0,0,0,0,0",
             "turvo":"+1,+1,0,0,0,0,+1,+1,+1"
             }

lgb_data_penalty={"abalone":0.1,
           "autompg":1.0,
           "autoprice":1.0,
           "boston":1.0,
           "california":1.0,
           "cpu":0.2,
           "crime":0.2,
           "redwine":1.0,
           "whitewine":1.0,
              "windsor":2,
             "turvo":3
                  }

lgb_data_margin={"abalone":0.0,
              "autompg":-1.0,
              "autoprice":0.0,
              "boston":0.0,
              "california":0.0,
              "cpu":0.0,
              "crime":0.0,
              "redwine":0.0,
              "whitewine":0.0,
             "windsor":-0.5,
             "turvo":-0.2
                 }


data_penalty={"abalone":0.1,
           "autompg":1.0,
           "autoprice":1.0,
           "boston":1.0,
           "california":1.0,
           "cpu":0.2,
           "crime":0.2,
           "redwine":1.0,
           "whitewine":1.0,
              "windsor":2,
             "turvo":3
              }

data_margin={"abalone":0.0,
              "autompg":-1.0,
              "autoprice":0.0,
              "boston":0.0,
              "california":2.0,
              "cpu":0.0,
              "crime":0.0,
              "redwine":0.0,
              "whitewine":0.0,
             "windsor":-0.5,
             "turvo":-0.2
             }