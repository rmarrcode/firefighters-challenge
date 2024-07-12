# Build czml files incrementally and algorthmically

import json

class Builder():

    packets = None
    config = None
    start = None
    finish = None
    
    def __init__(self):
        self.packets = {}
        self.packets['document'] = {"id" : "document", "name" : "Untitled", "version" : "1.0" }

    def dumps(self):
        return json.dumps(list(self.packets.values()))
    
    def name(self, name):
        self.packets['document']['name'] = name

    def configure(self, config):
        self.config = config

    def clock(self, start, finish):
        self.start=start
        self.finish=finish
        start_string = start.isoformat().replace('+00:00', 'Z')
        finish_string = finish.isoformat().replace('+00:00', 'Z')
        duration_string = start_string + "/" + finish_string
        self.packets['document']['clock'] = { "interval" : duration_string,
                                     "currentTime" : start_string,
                                     "multiplier" : 1.0,
                                     "range" : "LOOP_STOP",
                                     "step" : "SYSTEM_CLOCK_MULTIPLIER" }
        
    def add_ground_entity(self, unique, model, trajectory, description, minimum_pixel_size = 0):
        # trajectory is [ (datetime, longitude, latitude), ... ]
        epoch = trajectory[0][0].isoformat().replace('+00:00', 'Z')
        start = epoch
        finish = trajectory[-1][0].isoformat().replace('+00:00', 'Z')
        # encoded trajectory is [ seconds, longitude, latitude), ... ]
        encoded_trajectory = [((tuple[0] - trajectory[0][0]).total_seconds(), tuple[1], tuple[2], 0) for tuple in trajectory]
        # flat trajectory is all the right info in a single flat list
        flat_trajectory = []
        for row in encoded_trajectory: flat_trajectory.extend(row)
        if self.config and "model_directory" in self.config:
            model = self.config["model_directory"] + "/" + model
        self.packets[unique] = {"id" : unique, 
                            "name" : unique, 
                            "availability": start + "/" + finish, # "1999-09-27T20:55:01Z/1999-09-28T20:55:01Z"
                            "model" : {
                                "gltf" : model, 
                                "scale": 1.0,
                                "heightReference" : "CLAMP_TO_GROUND",
                                "minimumPixelSize" : minimum_pixel_size,
                            },
                            "position": { "interpolationAlgorithm" : "LINEAR", 
                                         "interpolationDegree": 1,
                                         "epoch": epoch,
                                         "cartographicDegrees" : flat_trajectory },
                            "orientation": { "velocityReference" : "#position" } 
                            }
        if not self.start or self.start > trajectory[0][0]: self.start = trajectory[0][0] # this track starts earlier
        if not self.finish or self.finish < trajectory[-1][0]: self.finish = trajectory[-1][0] # this track ends later
        self.clock(self.start, self.finish)




