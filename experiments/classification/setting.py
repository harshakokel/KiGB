data_target ={"adult":"income",
              "australia":"OUT",
              "car":"class",
              "cleveland":"unhealthy",
              "ljubljana":"Class",
              "FICO":"RiskPerformance_Bad"}


data_advice={"adult":"0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
             "australia":"0,0,0,0,0,0,+1,+1,+1,+1,0,0,0,0",
             "car":"-1,-1,0,+1,0,+1",
             "cleveland":"+1,+1,-1,+1,+1,+1,+1,-1,+1,+1,0,+1,0",
             "ljubljana":"+1,0,+1,+1,-1,+1,0,-1,0,0,0,0,0",
             "FICO": "-1,-1,-1,-1,-1,+1,+1,-1,-1,-1,-1,0,+1,0,-1,+1,+1,+1,+1,0,0,+1,0"}

data_penalty={"adult":2.0,
              "australia": 2.0,
              "car":0.5,
              "cleveland":1,
              "ljubljana":1,
              "FICO":2}


data_margin={"adult":0.0,
             "australia":-0.3,
             "car":0.3,
             "cleveland":-2,
             "ljubljana":-0.5,
             "FICO":-0.3}

lgb_penalty={"adult":2.0,
           "australia": 1,
           "car":1,
           "cleveland":0.5,
           "ljubljana":1,
            "FICO":1}

lgb_margin={"adult":0.2,
              "australia":-0.5,
              "car":-0.2,
              "cleveland":-0.2,
              "ljubljana":0,
             "FICO":0}