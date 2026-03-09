  use ort::session::Session;                                                                                                                                     
  use ort::session::builder::GraphOptimizationLevel;        
  use ort::ep::{WebGPU, CPU};
  use ort::value::TensorRef;
  use std::thread;                                                                                                                                               
   
  fn create_session(path: &str) -> Session {                                                                                                                     
      Session::builder()                                    
          .unwrap()
          .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()                                                                                      
          .with_execution_providers([                                                                                                                            
              WebGPU::default().build(),                                                                                                                         
              CPU::default().build().error_on_failure(),                                                                                                         
          ]).unwrap()                                       
          .commit_from_file(path)
          .unwrap()
  }                                                                                                                                                              
   
  fn main() {                                                                                                                                                    
      ort::init().with_name("webgpu_test").commit();        

      let model_path = std::env::args()
          .nth(1)
          .expect("Usage: ort-webgpu-thread-crash <path-to-onnx-model>");
                                                                                                                                                                 
      println!("=== Creating persistent sessions ===");
      let mut session1 = create_session(&model_path);                                                                                                                
      println!("Persistent session 1 created");             
      let mut session2 = create_session(&model_path);
      println!("Persistent session 2 created");                                                                                                                  
                                                                                                                                                                 
      println!("\n=== Running inference on persistent sessions + creating new ones concurrently ===");                                                           
                                                                                                                                                                 
      let input_shape: Vec<i64> = session1.inputs()[0]
          .dtype()
          .tensor_shape()
          .expect("Model input is not a tensor")
          .iter()
          .map(|&d| if d < 0 { 1 } else { d })
          .collect();                                                                                                                                            
                                                                                                                                                                 
      let total_elements: usize = input_shape.iter().product::<i64>() as usize;                                                                                  
      println!("Model input shape: {:?} ({} elements)", input_shape, total_elements);                                                                            
                                                                                                                                                                 
      let handles: Vec<_> = (0..5)                                                                                                                               
          .map(|i| {
              let path = model_path.clone();                                                                                                                     
              let shape = input_shape.clone();              
              let n = total_elements;
              thread::spawn(move || {                                                                                                                            
                  println!("Thread {i}: creating session and running inference...");
                  let mut session = create_session(&path);                                                                                                           
                  let data = vec![0.0f32; n];                                                                                                                    
                  let tensor = TensorRef::from_array_view((shape, &*data)).unwrap();
                  for round in 0..10 {
                      let _ = session.run(ort::inputs![tensor.view()]);                                                                                 
                      println!("Thread {i}: inference round {round} done");                                                                                      
                  }                                                                                                                                              
                  println!("Thread {i}: completed");                                                                                                             
              })                                                                                                                                                 
          })
          .collect();                                                                                                                                            
                                                            
      // also run inference on persistent sessions simultaneously
      let data = vec![0.0f32; total_elements];
      let tensor = TensorRef::from_array_view((input_shape, &*data)).unwrap();
      for round in 0..10 {
          let _ = session1.run(ort::inputs![tensor.view()]);
          let _ = session2.run(ort::inputs![tensor.view()]);                                                                                            
          println!("Main thread: inference round {round} done");
      }                                                                                                                                                          
                                                            
      for h in handles {                                                                                                                                         
          h.join().unwrap();                                
      }                                                                                                                                                          
                                                                                                                                                                 
  }                                          
                                                            